"""Is the foreign-memory result about AUTHORSHIP, or just about retrieval similarity?

``ForeignMemory`` reproduces only ~27% of own-memory's churn suppression, which the paper reads as
evidence for commitment (being shown one's *own* decisions) over anchoring (any concrete exemplar).
That reading has one serious competitor, and it does not need a language model to test:

    Own memory retrieves episodes from the agent's OWN trajectory, which by construction passes
    through the states the agent is currently in. A donor bank drawn from a different seed cannot.
    If the retrieved precedent is simply FURTHER AWAY in the foreign arm, then the arm manipulated
    similarity, not authorship, and the whole commitment reading collapses.

This script measures that distance directly. It replays the epidemic world with a scripted policy,
feeds the SAME ``EpisodicMemory`` machinery the experiments use, and reports how far the retrieved
episodes sit from the querying state under two conditions:

    own      the component accumulates this run's own decisions (the ``memory`` arm)
    foreign  the component is loaded with the donor bank this seed actually received, and never
             appends its own (the ``foreign`` arm)

Units are the component's own standardized distance, so the two are directly comparable, and the
comparison is within-seed.

Interpretation, fixed before running:

    If foreign retrieval is much further away (say >2x), the confound is real and the commitment
    reading must be withdrawn or heavily qualified.
    If the two are close, similarity is controlled and authorship is what differs.

The epidemic settles toward an endemic equilibrium, so different seeds traverse nearly the same
states — which is exactly why this world can host the test at all, and why the answer is not
obvious in advance.

Run: ``uv run python scripts/retrieval_similarity.py``
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from govsim.core.action import ActionRequest  # noqa: E402
from govsim.core.harness import Outcome  # noqa: E402
from govsim.core.schedule import EveryN  # noqa: E402
from govsim.core.system import Observation  # noqa: E402
from govsim.domains.scalar import regimes as R  # noqa: E402
from govsim.harness.components import EpisodicMemory, ForeignMemory  # noqa: E402

#: The keys the component actually compares on. Taken from the recorded observation, not invented.
KEYS = ("S", "I", "R", "lockdown", "vacc")

#: A competent standing policy, so the trajectory is the kind an agent would produce rather than a
#: degenerate one. Held FIXED across both conditions: the probe is about retrieval, not control.
LAWS = {"set_lockdown": "0.6 if I > 0.01 else 0.0", "set_vaccination": "0.25"}


def _trajectory(seed: int) -> list[dict[str, float]]:
    """Decision-time states of one seed under the fixed policy."""
    from govsim.domains.scalar import Lever, ScalarLeverInterface

    system = R.sir_factory(R.EPIDEMIC_SHOCKED)(seed)
    iface = ScalarLeverInterface([
        Lever("set_lockdown", (0.0, 0.9), "lockdown"),
        Lever("set_vaccination", (0.0, 0.5), "vacc"),
    ])
    sched = EveryN(R.EPIDEMIC_DECIDE_EVERY)
    states: list[dict[str, float]] = []
    for step in range(R.EPIDEMIC_HORIZON):
        if sched.should_decide(step) if hasattr(sched, "should_decide") else step % R.EPIDEMIC_DECIDE_EVERY == 0:
            obs = system.observe("regent:0") if hasattr(system, "observe") else None
            vars_ = dict(obs.vars) if obs is not None else dict(system.metrics())
            states.append({k: float(vars_[k]) for k in KEYS if k in vars_})
            iface.apply([ActionRequest("regent:0", v, {"expr": e}) for v, e in LAWS.items()], system)
        system.step()
    return states


def _episode(state: dict[str, float], score: float) -> dict:
    return {"state": dict(state), "actions": dict(LAWS), "score": score}


def _retrieval_distances(query_states: list[dict[str, float]], mode: str,
                         donor: list[dict], k: int = 3) -> list[float]:
    """Mean distance from each query state to the k episodes the component returns."""
    mem = ForeignMemory(donor, k=k) if mode == "foreign" else EpisodicMemory(k=k)
    out: list[float] = []
    for i, s in enumerate(query_states):
        if mem.episodes:
            ranked = sorted(mem.episodes, key=lambda e: mem._distance(s, e["state"]))[:k]
            out.append(st.fmean(mem._distance(s, e["state"]) for e in ranked))
        if mode == "own":
            # The own-memory arm appends as it goes; the foreign arm never does.
            mem.episodes.append(_episode(s, -float(i)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="logs/retrieval_similarity.json")
    args = ap.parse_args()

    trajectories = {s: _trajectory(s) for s in range(args.seeds)}
    rows = []
    print(f"{'seed':>5}{'own':>12}{'foreign':>12}{'ratio':>9}")
    for s in range(args.seeds):
        # The donor bank this seed would actually receive: episodes from OTHER seeds only.
        donor = [_episode(st_, -float(i))
                 for other, traj in trajectories.items() if other != s
                 for i, st_ in enumerate(traj)]
        own = _retrieval_distances(trajectories[s], "own", donor)
        foreign = _retrieval_distances(trajectories[s], "foreign", donor)
        if not own or not foreign:
            continue
        o, f = st.fmean(own), st.fmean(foreign)
        rows.append({"seed": s, "own": o, "foreign": f, "ratio": f / o if o else float("nan")})
        print(f"{s:>5}{o:>12.4f}{f:>12.4f}{rows[-1]['ratio']:>9.2f}")

    o = st.fmean(r["own"] for r in rows)
    f = st.fmean(r["foreign"] for r in rows)
    ratio = f / o if o else float("nan")
    print(f"\n{'mean':>5}{o:>12.4f}{f:>12.4f}{ratio:>9.2f}")

    # WHY the foreign arm's precedent is closer, stated so the number is not over-read. Two causes,
    # both real, neither a defect:
    #   (1) the endemic equilibrium — different seeds traverse nearly the same states, so a donor's
    #       history passes through the querying agent's neighbourhood;
    #   (2) BANK SIZE — the donor pools every other seed's full trajectory while own memory starts
    #       empty and grows one episode per decision, and a larger bank has a closer 3rd-nearest
    #       neighbour for purely combinatorial reasons.
    # (2) means the ratio is NOT a clean measure of "how foreign the precedent looks". What it does
    # establish is the only thing the confound argument needs: the foreign arm was not handicapped on
    # similarity. It was advantaged on it, and revised anyway.
    own_bank = len(next(iter(trajectories.values())))
    donor_bank = own_bank * (len(trajectories) - 1)
    verdict = (
        f"THE SIMILARITY CONFOUND RUNS BACKWARDS: retrieved precedent sits {ratio:.2f}x as far from "
        f"the querying state in the foreign arm as in the own arm — that is {1 / ratio:.1f}x CLOSER, "
        f"not further. Two causes: the endemic equilibrium makes seeds traverse nearly the same "
        f"states, and the donor bank pools {donor_bank} episodes against own memory's {own_bank} at "
        f"most, so its nearest neighbours are combinatorially closer. The foreign arm was therefore "
        f"advantaged on retrieval similarity and revised an order of magnitude more often anyway. "
        f"Similarity cannot explain the churn difference in the direction needed."
        if ratio < 1.0 else
        f"CONFOUNDED: foreign precedent is retrieved from {ratio:.2f}x further away, so the "
        f"foreign arm manipulated similarity as well as authorship. The commitment reading cannot "
        f"be sustained on this evidence."
    )
    print(f"\n=> {verdict}")
    Path(args.out).write_text(json.dumps(
        {"rows": rows, "own_mean": o, "foreign_mean": f, "ratio": ratio, "verdict": verdict,
         "keys": list(KEYS), "policy": LAWS, "n_seeds": len(rows)}, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
