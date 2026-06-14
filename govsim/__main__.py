"""
The govsim CLI — the real entry point the README always promised (the old ``__main__.py`` was
empty; doc-02 §3.1). Domain-agnostic: it runs any registered experiment through the core Runner.

    python -m govsim list
    python -m govsim run cubic_stabilization
    python -m govsim run cubic_nonlinear --seeds 0 1 2 --horizon 300 --store logs/runs --plot

Experiments register themselves in ``govsim.experiments`` (one function + one ``@register``).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from typing import Sequence

from govsim.core.runner import Runner
from govsim.core.result_store import ResultStore
from govsim import experiments


def _cmd_list(_args: argparse.Namespace) -> int:
    names = experiments.available()
    if not names:
        print("no experiments registered")
        return 0
    print("Available experiments:")
    for name in names:
        exp = experiments.get(name)
        print(f"  {name:<22} — H[{exp.hypothesis.id}]: {exp.hypothesis.claim}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        exp = experiments.get(args.experiment)
    except KeyError as e:
        print(e, file=sys.stderr)
        return 2

    if args.seeds:
        exp.seeds = list(args.seeds)
    if args.horizon:
        exp.horizon = args.horizon

    store = ResultStore(args.store) if args.store else None
    try:
        records = Runner(result_store=store).run(exp)
    except (RuntimeError, KeyError, ValueError) as e:
        print(f"run failed: {e}", file=sys.stderr)
        return 1

    print(f"=== {exp.name}  (H[{exp.hypothesis.id}], baseline: {exp.hypothesis.baseline}) ===")
    for rec in records:
        for rid, score in rec.score.items():
            comps = rec.components[rid]
            comp_str = ", ".join(f"{k}={v:.4f}" for k, v in comps.items())
            term = f"  [terminated@{rec.terminated_at_step}]" if rec.terminated_at_step is not None else ""
            print(f"  seed={rec.seed:<4} {rid}: score={score:.4f}  ({comp_str}){term}")

    for rid in exp.regents:
        scores = [rec.score[rid] for rec in records]
        if len(scores) > 1:
            mean = statistics.mean(scores)
            std = statistics.pstdev(scores)
            print(f"  {rid}: mean score={mean:.4f}  std={std:.4f}  (n={len(scores)} seeds)")

    if store is not None:
        print(f"  stored {len(records)} run(s) in {store.db_path}")
        if args.plot:
            rows = store.query(experiment=exp.name)
            out = store.plot(rows[0]["run_id"])
            print(f"  plotted run {rows[0]['run_id']} -> {out}")
    elif args.plot:
        print("  (--plot needs --store to persist series first)", file=sys.stderr)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from govsim.analysis import compare, infer_lower_is_better, metric_by_seed, collapse_summary

    try:
        exp_a = experiments.get(args.exp_a)
        exp_b = experiments.get(args.exp_b)
    except KeyError as e:
        print(e, file=sys.stderr)
        return 2
    if args.seeds:
        exp_a.seeds = list(args.seeds)
        exp_b.seeds = list(args.seeds)

    metric = args.metric or exp_a.hypothesis.primary_metric
    lower = infer_lower_is_better(metric)
    if args.higher_better:
        lower = False
    if args.lower_better:
        lower = True

    try:
        recs_a = Runner().run(exp_a)
        recs_b = Runner().run(exp_b)
    except (RuntimeError, KeyError, ValueError) as e:
        print(f"compare failed: {e}", file=sys.stderr)
        return 1

    a_by = metric_by_seed(recs_a, metric)
    b_by = metric_by_seed(recs_b, metric)
    res = compare(a_by, b_by, lower_is_better=lower)

    arrow = "lower-is-better" if lower else "higher-is-better"
    print(f"=== compare '{args.exp_a}' (A) vs '{args.exp_b}' (B) on '{metric}' ({arrow}) ===")
    for s in res["seeds"]:
        print(f"  seed={s:<4} A={a_by[s]:.4f}  B={b_by[s]:.4f}  A-B={a_by[s]-b_by[s]:+.4f}")
    print(f"  paired mean(A-B)={res['point_estimate']:+.4f}  95% CI=[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]  (n={res['n']})")
    if res["a_better_than_b"]:
        print(f"  VERDICT: A beats B (CI excludes 0 on the {arrow} side).")
    elif res["excludes_zero"]:
        print(f"  VERDICT: B beats A (CI excludes 0 against A).")
    else:
        print("  VERDICT: no significant difference (CI includes 0).")
    for label, recs in (("A", recs_a), ("B", recs_b)):
        cs = collapse_summary(recs)
        if cs["n_terminated"]:
            print(f"  [{label}] {cs['n_terminated']}/{cs['n_runs']} runs collapsed early (seeds {cs['terminated_seeds']}); worst score {cs['worst_score']:.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="govsim", description="Experiment machine for LLM regents of complex systems.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list registered experiments").set_defaults(func=_cmd_list)

    run = sub.add_parser("run", help="run a registered experiment")
    run.add_argument("experiment", help="experiment name (see `govsim list`)")
    run.add_argument("--seeds", type=int, nargs="+", help="override the experiment's seeds")
    run.add_argument("--horizon", type=int, help="override the horizon")
    run.add_argument("--store", help="persist runs to this ResultStore directory (sqlite + artifacts)")
    run.add_argument("--plot", action="store_true", help="plot the first run's series (needs --store)")
    run.set_defaults(func=_cmd_run)

    cmp = sub.add_parser("compare", help="paired bootstrap comparison of two experiments (A vs B)")
    cmp.add_argument("exp_a", help="experiment A (e.g. the harnessed LLM regent)")
    cmp.add_argument("exp_b", help="experiment B (e.g. the trace-less OPRO baseline)")
    cmp.add_argument("--metric", help="metric to compare (default: A's hypothesis.primary_metric)")
    cmp.add_argument("--seeds", type=int, nargs="+", help="shared seed list for the paired design")
    cmp.add_argument("--higher-better", action="store_true", help="force higher-is-better orientation")
    cmp.add_argument("--lower-better", action="store_true", help="force lower-is-better orientation")
    cmp.set_defaults(func=_cmd_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Load .env so OPENAI_*/AIRI_KEY/GOVSIM_LLM_* reach os.environ for live/cache LLM runs.
    # (No-op + harmless on CI: replay/scripted experiments never read a key.)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # pragma: no cover - dotenv optional at runtime
        pass
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
