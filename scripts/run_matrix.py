"""
Run the experiment matrix (arms x models x seeds) into a ResultStore.

The unit of work is one (model, arm) pair; seeds run inside it. Everything is persisted per pair,
so an interrupted sweep resumes rather than restarts — and because the LLM layer is a cache/replay
tape, a resumed pair costs nothing for the calls it already made.

    uv run python scripts/run_matrix.py --arms epidemic --seeds 20 --model gemini-3.5-flash-lite
    uv run python scripts/run_matrix.py --arms epidemic --models gemini-3.5-flash-lite gemma-4-31b-it

Progress goes to stdout one line per completed pair, terse enough to watch and specific enough to
tell a rate-limit stall from a crash.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from govsim.core.result_store import ResultStore
from govsim.core.runner import Runner

# --- arm groups ------------------------------------------------------------------------------

FACTORIAL = [
    "epidemic_llm_bare",
    "epidemic_llm_trace",
    "epidemic_llm_outcome",
    "epidemic_llm_memory",
    "epidemic_llm_trace_outcome",
    "epidemic_llm_trace_memory",
    "epidemic_llm_outcome_memory",
    "epidemic_llm_trace_outcome_memory",
]
REFERENCES = ["epidemic_frozen", "epidemic_best_fixed", "epidemic_switching", "epidemic_oracle"]
GROUPS = {
    # References first: they are key-free and instant, so a broken world config surfaces before
    # thousands of paid calls, not after.
    "epidemic": REFERENCES + FACTORIAL + ["epidemic_opro"],
    "epidemic-factorial": FACTORIAL,
    # The cross-model replication panel. The full 2^3 factorial is run once, on the primary model,
    # because component *attribution* needs all eight cells; the other models only have to answer
    # "does the effect replicate outside one model?", for which the three rungs no-harness /
    # outcome-only / full-harness suffice at a fraction of the call budget.
    "epidemic-lite": ["epidemic_llm_bare", "epidemic_llm_outcome",
                      "epidemic_llm_trace_outcome_memory"],
    "epidemic-refs": REFERENCES,
    "epidemic-critic": ["epidemic_llm_critic"],
    # The contextualization contrast: the naive channel is the control for the fixed one.
    "epidemic-ctx": ["epidemic_llm_outcome", "epidemic_llm_ctx_outcome"],
    "scalar": ["scalar_frozen", "scalar_oracle", "scalar_llm_full"],
    # The severity-tracking check: same three rungs, harsher break.
    "severe": ["severe_frozen", "severe_best_fixed", "severe_switching",
               "severe_llm_bare", "severe_llm_outcome", "severe_llm_trace_outcome_memory"],
}


def _needs_llm(arm: str) -> bool:
    return "llm" in arm or "opro" in arm


def run_pair(model: str, arm: str, seeds: list[int], store: ResultStore, workers: int = 1) -> dict:
    """Run one (model, arm) over all seeds. Returns a summary row; never raises.

    Seeds run concurrently when ``workers > 1``. Two things make that safe rather than merely fast:
    each worker calls ``experiments.get(arm)`` for itself, so no regent or harness object (and in
    particular no ``EpisodicMemory``) is shared across threads; and nothing is written to the store
    until every seed is back, because the sqlite connection belongs to the calling thread. The work
    is almost entirely waiting on the network, so the speedup is close to linear until the
    provider's rate limit binds and the client's backoff takes over.
    """
    os.environ["OPENAI_MODEL"] = model
    # Imported here, AFTER the env is set: experiment factories read OPENAI_MODEL at construction,
    # so the registry must be consulted per pair rather than captured once at import.
    from govsim import experiments

    t0 = time.time()

    def one_seed(seed: int):
        exp = experiments.get(arm)  # a FRESH regent + harness per seed
        exp.seeds = [seed]
        return Runner().run(exp)

    try:
        if workers <= 1:
            exp = experiments.get(arm)
            exp.seeds = list(seeds)
            records = Runner().run(exp)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                nested = list(pool.map(one_seed, seeds))
            records = [r for group in nested for r in group]
        for rec in records:  # single-threaded persistence
            store.add(rec)
    except Exception as e:  # noqa: BLE001 - one bad pair must not kill a multi-hour sweep
        print(f"FAILED  {model:<26} {arm:<36} {type(e).__name__}: {str(e)[:160]}", flush=True)
        traceback.print_exc(file=sys.stderr)
        return {"model": model, "arm": arm, "error": f"{type(e).__name__}: {e}"}

    # The PRIMARY metric is the full horizon: a post-break-only window rewards passivity, and
    # arms that behave differently BEFORE the break also arrive at it in different states, so a
    # post-window number is confounded by path dependence on top of that.
    losses = [r.components["regent:0"].get("loss", float("nan")) for r in records]
    finite = [v for v in losses if v == v and abs(v) != float("inf")]
    calls = sum(len(r.llm_io) for r in records)
    dt = time.time() - t0
    print(
        f"done    {model:<26} {arm:<36} n={len(records):<3} "
        f"loss mean={statistics.mean(finite) if finite else float('nan'):>9.4f} "
        f"sd={statistics.pstdev(finite) if len(finite) > 1 else 0.0:>7.4f} "
        f"llm_calls={calls:<5} {dt:>6.1f}s",
        flush=True,
    )
    return {
        "model": model, "arm": arm, "n": len(records),
        "loss_mean": statistics.mean(finite) if finite else None,
        "loss_sd": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
        "per_seed": {r.seed: r.components["regent:0"].get("loss") for r in records},
        "per_seed_post": {r.seed: r.components["regent:0"].get("post_loss") for r in records},
        "llm_calls": calls, "seconds": round(dt, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["epidemic"],
                    help=f"arm names, or a group: {sorted(GROUPS)}")
    ap.add_argument("--models", nargs="+", default=["gemini-3.5-flash-lite"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--store", default="logs/runs")
    ap.add_argument("--summary", default="logs/matrix_summary.json")
    ap.add_argument("--workers", type=int, default=1,
                    help="seeds to run concurrently within an arm (I/O bound; 4-6 is a good default)")
    args = ap.parse_args()

    arms: list[str] = []
    for a in args.arms:
        arms.extend(GROUPS.get(a, [a]))
    seeds = list(range(args.seeds))
    store = ResultStore(args.store)

    print(f"# matrix: {len(args.models)} model(s) x {len(arms)} arm(s) x {len(seeds)} seeds "
          f"-> {store.db_path} (workers={args.workers})", flush=True)
    print(f"# models: {', '.join(args.models)}", flush=True)

    rows: list[dict] = []
    for model in args.models:
        for arm in arms:
            # Key-free arms are model-independent; running them once per model would just
            # duplicate identical rows and muddy the aggregation.
            if not _needs_llm(arm) and any(r.get("arm") == arm for r in rows):
                continue
            rows.append(run_pair(model, arm, seeds, store, workers=args.workers))
            Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
            Path(args.summary).write_text(json.dumps(rows, indent=2), encoding="utf-8")

    failed = [r for r in rows if "error" in r]
    print(f"\n# complete: {len(rows) - len(failed)}/{len(rows)} pairs ok, "
          f"{sum(r.get('llm_calls', 0) for r in rows)} LLM calls total", flush=True)
    for r in failed:
        print(f"#   FAILED {r['model']} / {r['arm']}: {r['error']}", flush=True)
    print(f"# summary -> {args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
