"""
Export the model-call tape for exactly the runs a paper reports, and nothing else.

    uv run python scripts/export_tape.py --store logs/runs_v2 --out artifacts/tape

The papers promise that every reported run replays byte-for-byte with no key and no network. That
promise is only kept if the tape ships, and the working cache does not qualify: it accumulates every
call ever made, including abandoned runs, superseded prompt versions, and models that never made it
into a table. Shipping it would be both large and misleading — a reader could not tell which entries
back the results.

So the export is driven by the store: read each run's recorded LLM I/O, take the cache key each call
recorded, and copy across only those entries. What lands in ``--out`` is a minimal, self-contained
tape whose every file is load-bearing, plus a manifest saying which runs it covers.

Verify an export with:

    GOVSIM_LLM_MODE=replay GOVSIM_LLM_CACHE=artifacts/tape uv run python -m govsim run <arm>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from govsim.core.result_store import ResultStore


def _key_for(call: dict) -> str | None:
    """The cache key for one recorded call, read from the record rather than reconstructed.

    Reconstructing it from the I/O log does not work and the first version of this script proved
    it: the log keeps the messages and the response but not the tool schemas or ``max_tokens``,
    both of which enter the key, so every reconstructed key missed and the export produced an empty
    tape while cheerfully reporting 120 runs covered. ``CachingReplayClient`` now stamps the key it
    used onto the response and the regents record it, which is also what makes a replay miss
    diagnosable.

    Runs recorded before that change have no key. They are reported, not skipped silently — re-run
    them (every call is a cache hit, so it costs seconds) and the key will be there.
    """
    return call.get("cache_key") or None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", nargs="+", required=True)
    ap.add_argument("--cache", default="logs/llm_cache")
    ap.add_argument("--out", default="artifacts/tape")
    args = ap.parse_args()

    cache_dir = Path(args.cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    covered: dict[str, int] = {}
    copied, missing, unkeyable = 0, 0, 0
    for store_path in args.store:
        store = ResultStore(store_path)
        for row in store.query():
            path = row.get("llm_io_path")
            if not path or not Path(path).exists():
                continue
            calls = json.loads(Path(path).read_text(encoding="utf-8"))
            if not calls:
                continue
            covered[row["experiment"]] = covered.get(row["experiment"], 0) + 1
            for call in calls:
                key = _key_for(call)
                if key is None:
                    unkeyable += 1
                    continue
                src = cache_dir / f"{key}.json"
                dst = out / f"{key}.json"
                if dst.exists():
                    continue
                if src.exists():
                    shutil.copy2(src, dst)
                    copied += 1
                else:
                    missing += 1

    manifest = {
        "source_stores": list(args.store),
        "source_cache": str(cache_dir),
        "entries": copied,
        "runs_by_experiment": covered,
        "keys_not_found_in_cache": missing,
        "calls_without_reconstructable_key": unkeyable,
        "replay_with": "GOVSIM_LLM_MODE=replay GOVSIM_LLM_CACHE=" + str(out),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"exported {copied} tape entries covering {sum(covered.values())} runs "
          f"across {len(covered)} experiments -> {out}")
    for exp, n in sorted(covered.items()):
        print(f"  {exp:<40} {n} run(s)")
    if missing or unkeyable:
        print(f"\n[!] {missing} key(s) not present in {cache_dir}, "
              f"{unkeyable} call(s) had no reconstructable key. The export is INCOMPLETE and a "
              f"replay against it will raise on the missing entries.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
