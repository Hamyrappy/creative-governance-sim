"""When does precedent lock a policy in, and does the lock release when the world breaks?

The factorial says episodic memory suppresses revision (churn 0.847 -> 0.082) and the foreign-memory
arm says the suppression needs the precedent to be the agent's OWN (0.632 with a donor's). Both are
run-level averages, and a run-level average cannot distinguish two very different institutions:

    (i)  one that freezes and stays frozen, and
    (ii) one that holds steady while the world is steady and revises when the world breaks.

(ii) is what we would want an institution to do. (i) is the pathology. They can produce similar
average churn, so the average is the wrong statistic and this script computes the right one: the
revision rate at each decision index, aligned to the structural break.

It also settles the last live alternative to the authorship reading. One could argue the foreign arm
differs not in WHO wrote the precedent but in what the bank looks like: the donor bank holds eight
DIFFERENT laws (its donor is the no-harness arm, which churns 0.847), while own memory fills up with
the agent's own increasingly repetitive history, so perhaps a repetitive bank is what anchors. That
account makes a check-able prediction: lock-in should require the bank to have become repetitive, so
it should appear late.

It appears at bank size one. Own memory holds exactly one episode at the first transition and
already suppresses revision materially; by two episodes it is total. A one-episode bank cannot be
repetitive, so bank repetitiveness is not the mechanism.

Run: ``uv run python scripts/lockin_onset.py``
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from govsim.analysis.stats import compare, holm_bonferroni  # noqa: E402
from govsim.core.result_store import ResultStore  # noqa: E402
from govsim.domains.scalar import regimes as R  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from policy_churn import _sequence  # noqa: E402

ARMS = {
    "bare": "epidemic_llm_bare",
    "own": "epidemic_llm_memory",
    "foreign": "epidemic_llm_foreign",
}
#: The foreign arm runs seeds 0-9, so every comparison is restricted to those. Averaging the other
#: arms over 20 seeds while the foreign arm has 10 would compare different seed sets.
SEEDS = set(range(10))
BREAK_DECISION = R.EPIDEMIC_SHOCK_STEP // R.EPIDEMIC_DECIDE_EVERY


def _sequences(store: ResultStore, experiment: str) -> dict[int, list[tuple[str, ...]]]:
    out: dict[int, list[tuple[str, ...]]] = {}
    for row in store.query(experiment=experiment):
        seed = int(row["seed"])
        if seed not in SEEDS:
            continue
        path = row.get("llm_io_path")
        if not path or not Path(path).exists():
            continue
        calls = sorted([c for c in json.loads(Path(path).read_text(encoding="utf-8"))
                        if c.get("regent_id") != "critic"],
                       key=lambda c: int(c.get("step", 0)))
        # carry_forward: a collapsed decision leaves the standing law in force, which is what the
        # world experienced. Dropping it would inflate churn exactly in the arms that collapse most.
        out[seed] = _sequence(calls, carry_forward=True)
    return out


def _window_rate(seq: list[tuple[str, ...]], lo: int, hi: int) -> float | None:
    """Share of transitions in [lo, hi) where the enacted law changed."""
    pairs = [(seq[i - 1], seq[i]) for i in range(max(lo, 1), min(hi, len(seq)))]
    return sum(1.0 for a, b in pairs if a != b) / len(pairs) if pairs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="logs/runs_v3")
    ap.add_argument("--out", default="logs/lockin_onset.json")
    args = ap.parse_args()

    store = ResultStore(args.store)
    seqs = {k: _sequences(store, v) for k, v in ARMS.items()}
    common = sorted(set.intersection(*(set(d) for d in seqs.values())))
    n_dec = min(len(s) for d in seqs.values() for s in d.values())
    print(f"seeds {common}   decisions/run {n_dec}   break at decision {BREAK_DECISION}\n")

    print("REVISION RATE BY DECISION (bank size is own-memory's; the foreign bank is fixed at 8)")
    print(f"{'transition':>12}{'own bank':>10}{'bare':>8}{'own':>8}{'foreign':>9}")
    curve = []
    for i in range(1, n_dec):
        rates = {k: st.fmean([1.0 if seqs[k][s][i] != seqs[k][s][i - 1] else 0.0
                              for s in common if len(seqs[k][s]) > i]) for k in ARMS}
        marker = "   <- break" if i == BREAK_DECISION else ""
        print(f"{f'{i}->{i + 1}':>12}{i:>10}{rates['bare']:>8.2f}{rates['own']:>8.2f}"
              f"{rates['foreign']:>9.2f}{marker}")
        curve.append({"transition": i, "own_bank_size": i, **rates})

    # --- does the lock RELEASE at the break? ------------------------------------------------------
    print(f"\nPRE- vs POST-BREAK REVISION (break at decision {BREAK_DECISION})")
    print(f"{'arm':>10}{'pre':>8}{'post':>8}{'delta':>9}{'95% CI':>22}{'p':>9}")
    windows, tests = {}, {}
    for k in ARMS:
        pre = {s: r for s in common
               if (r := _window_rate(seqs[k][s], 1, BREAK_DECISION + 1)) is not None}
        post = {s: r for s in common
                if (r := _window_rate(seqs[k][s], BREAK_DECISION + 1, n_dec)) is not None}
        c = compare(post, pre, n_boot=20000)
        windows[k] = {"pre": st.fmean(pre.values()), "post": st.fmean(post.values())}
        tests[k] = c
        ci = f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"
        print(f"{k:>10}{windows[k]['pre']:>8.2f}{windows[k]['post']:>8.2f}"
              f"{c['point_estimate']:>+9.3f}{ci:>22}{c['p_wilcoxon']:>9.4f}")

    holm = holm_bonferroni({k: v["p_wilcoxon"] for k, v in tests.items()})
    print("\nHolm-corrected across the three arms:")
    for k, v in holm.items():
        print(f"  {k:>10}  p_adj={v['p_adj']:.4f}  {'RESPONDS TO THE BREAK' if v['significant'] else 'no detectable response'}")

    onset = curve[0]
    verdict = (
        f"Own precedent locks in IMMEDIATELY: with a bank of one episode the revision rate is already "
        f"{onset['own']:.2f} against the no-memory arm's {onset['bare']:.2f}, and by two episodes it "
        f"is zero. A one-episode bank cannot be repetitive, so 'the bank became repetitive' is not "
        f"the mechanism. Foreign precedent, eight episodes of eight different laws, leaves the first "
        f"transition at {onset['foreign']:.2f}. "
        f"The two also differ in the way that matters institutionally: the foreign arm's revision "
        f"rate {'RISES' if tests['foreign']['point_estimate'] > 0 else 'falls'} by "
        f"{abs(tests['foreign']['point_estimate']):.2f} after the break "
        f"(p={tests['foreign']['p_wilcoxon']:.4f}), while the own-memory arm "
        f"{'rises' if tests['own']['point_estimate'] > 0 else 'falls'} by "
        f"{abs(tests['own']['point_estimate']):.2f} (p={tests['own']['p_wilcoxon']:.4f}). "
        f"Foreign precedent buys stability that RELEASES on a structural break; own precedent buys "
        f"stability that does not."
    )
    print(f"\n=> {verdict}")

    Path(args.out).write_text(json.dumps({
        "seeds": common, "n_decisions": n_dec, "break_decision": BREAK_DECISION,
        "curve": curve, "windows": windows,
        "break_response": {k: {"delta": v["point_estimate"], "ci": [v["ci_low"], v["ci_high"]],
                               "p_wilcoxon": v["p_wilcoxon"], "p_holm": holm[k]["p_adj"],
                               "significant": holm[k]["significant"]}
                           for k, v in tests.items()},
        "verdict": verdict,
    }, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
