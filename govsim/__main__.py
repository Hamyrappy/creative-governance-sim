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
    records = Runner(result_store=store).run(exp)

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
