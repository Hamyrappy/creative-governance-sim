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
REFERENCES = ["epidemic_frozen", "epidemic_oracle"]
GROUPS = {
    # References first: they are key-free and instant, so a broken world config surfaces before
    # thousands of paid calls, not after.
    "epidemic": REFERENCES + FACTORIAL + ["epidemic_opro"],
    "epidemic-factorial": FACTORIAL,
    "epidemic-refs": REFERENCES,
    "epidemic-critic": ["epidemic_llm_critic"],
    "scalar": ["scalar_frozen", "scalar_oracle", "scalar_llm_full"],
}


def _needs_llm(arm: str) -> bool:
    return "llm" in arm or "opro" in arm


def run_pair(model: str, arm: str, seeds: list[int], store: ResultStore) -> dict:
    """Run one (model, arm) over all seeds. Returns a summary row; never raises."""
    os.environ["OPENAI_MODEL"] = model
    # Imported here, AFTER the env is set: experiment factories read OPENAI_MODEL at construction,
    # so the registry must be consulted per pair rather than captured once at import.
    from govsim import experiments

    t0 = time.time()
    try:
        exp = experiments.get(arm)
        exp.seeds = list(seeds)
        records = Runner(result_store=store).run(exp)
    except Exception as e:  # noqa: BLE001 - one bad pair must not kill a multi-hour sweep
        print(f"FAILED  {model:<26} {arm:<36} {type(e).__name__}: {str(e)[:160]}", flush=True)
        traceback.print_exc(file=sys.stderr)
        return {"model": model, "arm": arm, "error": f"{type(e).__name__}: {e}"}

    losses = [r.components["regent:0"].get("post_loss", float("nan")) for r in records]
    finite = [v for v in losses if v == v and abs(v) != float("inf")]
    calls = sum(len(r.llm_io) for r in records)
    dt = time.time() - t0
    print(
        f"done    {model:<26} {arm:<36} n={len(records):<3} "
        f"post_loss mean={statistics.mean(finite) if finite else float('nan'):>9.4f} "
        f"sd={statistics.pstdev(finite) if len(finite) > 1 else 0.0:>7.4f} "
        f"llm_calls={calls:<5} {dt:>6.1f}s",
        flush=True,
    )
    return {
        "model": model, "arm": arm, "n": len(records),
        "post_loss_mean": statistics.mean(finite) if finite else None,
        "post_loss_sd": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
        "per_seed": {r.seed: r.components["regent:0"].get("post_loss") for r in records},
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
    args = ap.parse_args()

    arms: list[str] = []
    for a in args.arms:
        arms.extend(GROUPS.get(a, [a]))
    seeds = list(range(args.seeds))
    store = ResultStore(args.store)

    print(f"# matrix: {len(args.models)} model(s) x {len(arms)} arm(s) x {len(seeds)} seeds "
          f"-> {store.db_path}", flush=True)
    print(f"# models: {', '.join(args.models)}", flush=True)

    rows: list[dict] = []
    for model in args.models:
        for arm in arms:
            # Key-free arms are model-independent; running them once per model would just
            # duplicate identical rows and muddy the aggregation.
            if not _needs_llm(arm) and any(r.get("arm") == arm for r in rows):
                continue
            rows.append(run_pair(model, arm, seeds, store))
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
