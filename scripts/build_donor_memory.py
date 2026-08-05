"""Build the donor episode bank for ``ForeignMemory`` from recorded runs.

``ForeignMemory`` shows an agent precedent it did not author, to separate two readings of the
lock-in result that presentation repairs could not distinguish: does *any* concrete exemplar anchor
the agent (anchoring), or specifically *its own* prior decisions (commitment)?

That only works if the donor episodes are genuinely foreign and genuinely comparable. This script
reconstructs, from a recorded arm's LLM-I/O artifacts, the same ``(state, actions, score)`` triples
that :class:`~govsim.harness.components.EpisodicMemory` would have stored, and writes one bank per
seed containing episodes from OTHER seeds only.

The donor arm should be a competent one governed under the SAME world and mandate — we use the
no-harness arm, because its policies are the ones an unaided agent actually produced, so the
precedent is realistic rather than idealised. Using the clairvoyant reference instead would turn the
component into a demonstration channel and test a different question.

Run: ``uv run python scripts/build_donor_memory.py --store logs/runs_v3 --arm epidemic_llm_bare``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from govsim.core.action import ActionSpace, VerbSpec  # noqa: E402
from govsim.core.llm.client import LLMResponse  # noqa: E402
from govsim.core.result_store import ResultStore  # noqa: E402
from govsim.harness.components import EpisodicMemory  # noqa: E402
from govsim.regents.llm_regent import parse_action_requests  # noqa: E402

SPACE = ActionSpace(
    verbs=[VerbSpec(name=v) for v in ("set_lockdown", "set_vaccination", "set_policy_rate")],
    context_vars=[],
)


#: The prompt line the Runner writes for every decision, e.g.
#: ``Current observation (step 20): S=0.813119, I=0.0178705, R=0.16901, lockdown=0.686272, ...``
#: Parsed rather than read from a structured field because the recorded I/O keeps only ``messages``,
#: and a component that retrieves against an EMPTY state is not a weaker version of the real one —
#: every distance becomes +inf, retrieval degenerates to arbitrary order, and the arm looks live
#: while carrying no similarity signal at all.
_OBS_RE = re.compile(r"Current observation \(step \d+\):\s*(.+)")
_KV_RE = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def _state_from_prompt(call: dict) -> dict[str, float]:
    text = "\n".join((m.get("content") or "") for m in (call.get("messages") or [])
                     if isinstance(m, dict))
    m = _OBS_RE.search(text)
    if not m:
        return {}
    return {k: float(v) for k, v in _KV_RE.findall(m.group(1))}


def _series(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        for line in fh:
            vals = line.strip().split(",")
            if len(vals) == len(header):
                rows.append({h: float(v) for h, v in zip(header, vals)})
    return rows


def episodes_for_seed(store: ResultStore, arm: str, seed: int) -> list[dict]:
    """Reconstruct one run's episodes in the shape ``EpisodicMemory`` stores them.

    The score must be computed the way the component computes it — the SUM of the metric values it
    does not exclude — or the donor bank would carry scores on a different scale from the ones a
    self-memory arm shows, and the comparison would confound provenance with magnitude.
    """
    out: list[dict] = []
    for row in store.query(experiment=arm):
        if int(row.get("seed", -1)) != seed:
            continue
        path = row.get("llm_io_path")
        if not path or not Path(path).exists():
            continue
        series_path = Path(str(path).replace("_llm_io.json", "_series.csv"))
        series = _series(series_path) if series_path.exists() else []
        calls = sorted(
            [c for c in json.loads(Path(path).read_text(encoding="utf-8"))
             if c.get("regent_id") != "critic"],
            key=lambda c: int(c.get("step", 0)),
        )
        for call in calls:
            reqs = parse_action_requests(
                LLMResponse(text=call.get("response_text") or "",
                            tool_calls=list(call.get("tool_calls") or [])), SPACE, "regent:0")
            if not reqs:
                continue
            state = _state_from_prompt(call)
            if not state:
                continue
            step = int(call.get("step", 0))
            row_at = next((r for r in series if int(r.get("t", -1)) == step), None)
            if row_at is None:
                # DROP rather than default to 0.0. The series begins at t=1, so every step-0
                # decision has no row; zero-filling them would put a fabricated "this policy scored
                # 0" into the donor bank — 12.5% of episodes on the first build — against a real
                # score range of 2.0-2.45. Foreign precedent would then look worse than self
                # precedent for a reason that has nothing to do with provenance, which is the one
                # thing this component is meant to isolate.
                continue
            score = sum(v for k, v in row_at.items()
                        if k not in EpisodicMemory._NON_STATE_KEYS
                        and not EpisodicMemory.is_scorer_only(k))
            out.append({
                "state": state,
                "actions": [{"verb": r.verb, "expr": str(r.payload.get("expr"))} for r in reqs],
                "score": float(score),
                "_donor_seed": seed,
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="logs/runs_v3")
    ap.add_argument("--arm", default="epidemic_llm_bare")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--k", type=int, default=8, help="episodes per donor bank")
    ap.add_argument("--out", default="logs/donor_memory.json")
    args = ap.parse_args()

    store = ResultStore(args.store)
    per_seed = {s: episodes_for_seed(store, args.arm, s) for s in range(args.seeds)}
    have = {s: len(v) for s, v in per_seed.items() if v}
    print(f"reconstructed episodes per seed: {have}")
    if not have:
        print("no episodes recovered — is the arm recorded in this store?")
        return 1

    # Each seed's bank comes from the NEXT seed's run, so no agent ever sees its own trajectory and
    # the donor is a different world realisation as well as a different agent history.
    banks: dict[str, list[dict]] = {}
    seeds = sorted(have)
    for i, s in enumerate(seeds):
        donor = seeds[(i + 1) % len(seeds)]
        banks[str(s)] = per_seed[donor][: args.k]
    Path(args.out).write_text(json.dumps(banks, indent=1), encoding="utf-8")
    sizes = {s: len(b) for s, b in banks.items()}
    print(f"wrote {args.out}: {len(banks)} banks, sizes {sorted(set(sizes.values()))}")

    # A bank whose episodes carry no state cannot be retrieved against, and EpisodicMemory's
    # distance would fall back to +inf for every candidate — the component would inject its first k
    # episodes in arbitrary order and look live while carrying no similarity signal at all.
    empty_state = sum(1 for b in banks.values() for e in b if not e["state"])
    total = sum(len(b) for b in banks.values())
    print(f"episodes with an empty state: {empty_state}/{total}"
          + ("   <-- retrieval will be degenerate; fix the reconstruction before running the arm"
             if empty_state else ""))
    # A fabricated score is worse than a missing episode: it puts "this policy achieved X" into the
    # bank when nothing achieved X, and the agent has no way to discount it.
    zero = sum(1 for b in banks.values() for e in b if e["score"] == 0.0)
    print(f"episodes with a zero score: {zero}/{total}"
          + ("   <-- these are fabricated; they must be dropped, not defaulted" if zero else ""))
    # No agent may see its own trajectory.
    leaks = [s for s, b in banks.items() if any(e.get("_donor_seed") == int(s) for e in b)]
    print(f"banks containing the agent's OWN episodes: {len(leaks)}"
          + (f" {leaks}   <-- the component is not foreign" if leaks else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
